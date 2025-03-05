from evox.core import Algorithm, Problem, vmap
from evox.utils import clamp
from functools import partial
import torch

def euclidean_distance(a, b):
    # Calculate the normalized Euclidean distance between two real-valued vectors
    return torch.sqrt(torch.sum((a - b)**2))


def relative_distance(a, b, n_dims, lb, ub):
    # Calculate relative distance normalized by the maximum possible distance in the space
    max_possible_dist = (n_dims ** 0.5) * (ub - lb)
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


def tournament(_idx, population, fitness, tournament_size):
    """
    Perform tournament selection.

    Args:
    _idx as a hack to make it work with vmap
    population (jnp.array): The current population.
    fitness (jnp.array): Fitness values of the population.
    tournament_size (int): Number of individuals in each tournament.

    Returns:
    int: Index of the tournament winner.
    """
    indices = torch.randint(size=(tournament_size,), low=0, high=population.shape[0])
    selected = fitness[indices]
    winner_idx = torch.argmin(selected, dim=0, keepdim=True)
    return torch.gather(indices, 0, winner_idx).squeeze()


def difference_function_tournament(parent1, tau, population, fitness, tournament_size, tau_max, minimization, n_dims, lb, ub, debug_flag=False):
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
    indices = torch.randint(size=(tournament_size,), low=0, high=population.shape[0])

    candidates = population[indices]
    candidate_fitness = fitness[indices]

    # Calculate all relative distances
    d_i = vmap(lambda c: relative_distance(parent1, c, n_dims, lb, ub), randomness="different")(candidates)
    
    # Print statistics of the distances if debug is enabled
    if debug_flag:
        print("Distance stats - Min: {} Max: {} Mean: {} Std: {}", 
              torch.min(d_i), torch.max(d_i), torch.mean(d_i), torch.std(d_i))
    
    D = difference_function(tau, tau_max, d_i, minimization)

    adjusted_fitness = candidate_fitness * D
    winner_index = indices[torch.argmin(adjusted_fitness)]
    
    return winner_index


def full_tournament(population, fitness, pref, tournament_size, tau_max, minimization, n_dims, lb, ub, debug_flag=False):
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

    # select first parents
    parent1_indices = vmap(
        partial(
            tournament, 
            population=population, 
            fitness=fitness, 
            tournament_size=tournament_size), randomness="different")(torch.arange(pop_size // 2))

    # select second parents based on preference dif function (for each parent1, select a parent2 using diff tourn with all pop)
    parent2_indices = vmap(
        partial(
            difference_function_tournament,
            population=population,
            fitness=fitness,
            tournament_size=tournament_size,
            tau_max=tau_max,
            minimization=minimization,
            n_dims=n_dims,
            lb=lb,
            ub=ub,
            debug_flag=debug_flag),randomness="different")(population[parent1_indices], pref)
    
    parents = torch.stack([population[parent1_indices], population[parent2_indices]], axis=1)
    parents_index = torch.stack([parent1_indices, parent2_indices], axis=1)

    return parents, parents_index


def geometric_crossover(parents):
    """
    Perform geometric crossover in continuous space.

    Args:
    parents (jnp.array): Pair of parents.

    Returns:
    tuple: Two offspring inside [p1,p2]
    """
    p1, p2 = parents
    alpha = torch.rand(size=(1,))
    offspring1 = alpha * p1 + (1-alpha) * p2
    offspring2 = (1-alpha) * p1 + alpha * p2
    return torch.stack([offspring1, offspring2], axis=0)


def non_geometric_crossover(parents):
    """
    Perform non-geometric (extension ray) crossover in continuous space.

    Args:
    parents (jnp.array): Pair of parents.

    Returns:
    tuple: Two offspring extending beyond p2 and p1 respectively
    """
    p1, p2 = parents
    direction = p2 - p1
    alpha = 1 #torch.rand(size=(1,)) mexico  # Control extension amount.. alpha = 1 would be the correct translation of my thesis
    offspring1 = p2 + alpha * direction      # Extend beyond p2
    offspring2 = p1 - alpha * direction      # Extend beyond p1
    return torch.stack([offspring1, offspring2], axis=0)


def crossover(pref, parents):
    """
    Choose between geometric and non-geometric crossover based on preference.

    Args:
    pref (int): Preference level determining crossover type.
    parents (jnp.array): Pair of parents -- (2, dim)

    Returns:
    tuple: Two offspring created by the selected crossover method.
    """
    return torch.where(
        pref == 1, # 4?
        non_geometric_crossover(parents),
        geometric_crossover(parents)
    )


def batch_crossover(pref, parents):
    """
    Apply crossover to multiple pairs of parents in parallel.

    Args:
    pref: Preference levels for each pair.
    parents: All parent pairs.

    Returns:
    torch.Tensor: All offspring created by crossover.
    """
    # crossover on multiple pairs of parents -- (n_pair, 2, dim)
    return vmap(crossover, randomness="different")(pref, parents)


class Pulse_real(Algorithm):
    def __init__(self, pop_size, dim, lb, ub, p_c, p_m, tournament_size=3, tau_max=3, debug=False):
        super().__init__()
        self.pop_size = pop_size
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.crossover = batch_crossover
        self.mutation = None #mutation
        self.n_offspring = self.pop_size
        assert self.n_offspring % 2 == 0, "n_offspring must be even"
        self.tournament_size = tournament_size
        self.tau_max = tau_max
        self.selection = partial(
            full_tournament,
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

        self.population = (
            torch.rand(size=(self.pop_size, self.dim)) * (ub - lb) + lb
        )

        self.contrib = torch.ones((4,)) / 4
        self.total_cross = torch.zeros((4,), dtype=int)
        self.succ_cross = torch.zeros((4,), dtype=int)
        self.parents = torch.empty(
            (self.n_offspring // 2, 2, self.dim), dtype=self.population.dtype
        )
        self.parents_index = torch.empty((self.n_offspring // 2, 2), dtype=int)
        self.offspring = torch.empty((self.n_offspring, self.dim), dtype=self.population.dtype)
        self.fitness = torch.empty((self.pop_size,), dtype=float)
        self.pref = torch.empty((self.n_offspring // 2,), dtype=int)
        
        

    def _is_succ_cross(self, parents, offspring, par_fit, off_fit):
        # return True if it is a successful crossover
        # successful means one of the offspring satisfy the following:
        # 1. better or equal than both parents
        # 2. different from both parents

        # parents, offspring --- (2, self.dim)
        # par_fit, off_fit ----- (2, )

        is_better = off_fit <= torch.min(par_fit)  # most problems are minimization, otherwise we have to pay attention

        # Small threshold for floating point comparison
        epsilon = 1e-6
        is_eq = torch.stack([
            torch.all(torch.abs(offspring[0] - parents[0]) < epsilon) | 
            torch.all(torch.abs(offspring[0] - parents[1]) < epsilon),
            torch.all(torch.abs(offspring[1] - parents[0]) < epsilon) | 
            torch.all(torch.abs(offspring[1] - parents[1]) < epsilon),
        ], dim=0) 
        # if any offspring satisfy: 1. not equal 2. better # big brain type shi
        return torch.any(~is_eq & is_better) 

    def _update_variables(self, off_fit):
        # Update the counts of successful and total crossovers. parents should come in the correct shape but just in case reshape 
        par_fit = self.fitness[self.parents_index]
        parents = self.parents.reshape(-1, 2, self.dim)
        offspring = self.offspring.reshape(-1, 2, self.dim)
        par_fit = par_fit.reshape(-1, 2)
        off_fit = off_fit.reshape(-1, 2)
        is_succ_cross = vmap(self._is_succ_cross)(parents, offspring, par_fit, off_fit)
        self.succ_cross[self.pref] += is_succ_cross
        self.total_cross[self.pref] += 1

    def _update_contrib(self):
        # Update the contribution values for each preference level.
        contrib = self.succ_cross / self.total_cross
        total = torch.sum(contrib)
        self.contrib = torch.where(
            total == 0, torch.ones((4,)) / 4, 0.1 + (0.6 * (contrib / total))
        )
        # Reset succ_cross and total_cross
        self.succ_cross = torch.zeros_like(self.succ_cross)
        self.total_cross = torch.zeros_like(self.total_cross)

    def init_step(self):
        self.fitness = self.evaluate(self.population)

    def step(self):
        # Generate new offspring through selection, crossover, and mutation.
        if self.debug:
            print("Current contribution values: {}", self.contrib)
        
        # choice a pref based on the current contribution
        # every 2 offsprings needs 1 pref
        pref = torch.multinomial(
            self.contrib, num_samples=self.n_offspring // 2, replacement=True
        )

        counts = torch.zeros(4, dtype=torch.int32)
        for p in pref:
            counts[p] += 1
        final_counts = counts
        percentages = final_counts / torch.sum(final_counts) * 100
        
        if self.debug:
            print("Preferences shape: {}, Percentages: {}",
                  pref.shape,
                  torch.column_stack((torch.arange(4), percentages)))
        
        # select a batch of parents
        parents, parents_index = self.selection(self.population, self.fitness, pref)
        if self.debug:
            print("Selected parents shape: {}", parents.shape)
            print("Parents index shape: {}", parents_index.shape)
        
        # reshape into pairs of two, and do crossover on each pair
        offspring = self.crossover(pref, parents)
        offspring = offspring.reshape(self.pop_size, self.dim)
        if self.debug:
            print("Offspring shape after crossover: {}", offspring.shape)

        # mutation
        #offspring = self.mutation(offspring)
        if self.debug:
            print("Offspring shape after mutation: {}", offspring.shape)
        
        # Ensure offspring are within bounds
        offspring = clamp(offspring, self.lb, self.ub) # this is pathetic, glued spaces is the way to go 
        self.pref = pref
        self.offspring = offspring
        self.parents = parents
        self.parents_index = parents_index

        fitness = self.evaluate(offspring) # tell

        # Process fitness values of offspring and update algorithm state - contributions of preferences.
        if self.debug:
            print("Fitness shape: {}", fitness.shape)
            print("Fitness statistics - Min: {} Max: {} Mean: {}",
                   torch.min(fitness),
                   torch.max(fitness),
                   torch.mean(fitness))
        
        state = self._update_variables(fitness)
        if self.debug:
            print("Updated successful crossovers: {}", state.succ_cross)
            print("Updated total crossovers: {}", state.total_cross)
        
        self._update_contrib(state)
        if self.debug:
            print("Updated contribution values: {} \n\n-------------------------\n", state.contrib)
        
        return state
    
    def record_step(self):
        """A callback function to record the information of the algorithm"""
        return {"pop": self.population, "fit": self.fitness}

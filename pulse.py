from evox import Algorithm, Problem, State, jit_class, operators, workflows, monitors
import jax.numpy as jnp
from jax import vmap, lax, random
from functools import partial


def hamming_distance(a, b):
    return jnp.sum(a != b)


def difference_function(tau, tau_max, d_i, minimization):
    if minimization:
        return 0.5 + (tau / tau_max) * (0.5 - d_i)
    else:
        return 0.5 + (tau / tau_max) * (d_i - 0.5)


def tournament(key, population, fitness, tournament_size):
    keys = random.split(key, tournament_size)
    indices = vmap(lambda k: random.randint(k, (), 0, population.shape[0]))(keys)
    selected = fitness[indices]
    winner_index = indices[jnp.argmin(selected)]
    return winner_index


def preference_tournament(key, parent1, population, fitness, tournament_size, tau, tau_max, minimization):
    keys = random.split(key, tournament_size)
    indices = vmap(lambda k: random.randint(k, (), 0, population.shape[0]))(keys)

    candidates = population[indices]
    candidate_fitness = fitness[indices]

    d_i = vmap(lambda c: hamming_distance(parent1, c) / parent1.shape[0])(candidates)
    D = difference_function(tau, tau_max, d_i, minimization)

    adjusted_fitness = candidate_fitness * D
    winner_index = indices[jnp.argmin(adjusted_fitness)]
    return winner_index


def full_tournament(key, population, fitness, pref, tournament_size, tau_max, minimization):
    pop_size = population.shape[0]
    keys = random.split(key, pop_size * 2)

    # Select first parents
    parent1_indices = vmap(partial(tournament, population=population, fitness=fitness, tournament_size=tournament_size))(keys[:pop_size])

    # Select second parents based on preference
    parent2_indices = vmap(partial(preference_tournament,
                                   population=population,
                                   fitness=fitness,
                                   tournament_size=tournament_size,
                                   tau_max=tau_max,
                                   minimization=minimization))(keys[pop_size:], population[parent1_indices], pref)

    parents = jnp.stack([population[parent1_indices], population[parent2_indices]], axis=1)
    parents_index = jnp.stack([parent1_indices, parent2_indices], axis=1)

    return parents, parents_index


def extension_ray_crossover(key, parents):
    # create offsprings by extending from [p1,p2[, second offspring by extending from [p2,p1[
    p1, p2 = parents
    commonality = p1 == p2
    offspring1 = jnp.where(commonality, ~p2, p2)
    offspring2 = jnp.where(commonality, ~p1, p1)
    return offspring1, offspring2


def one_point_crossover(key, parents):
    # create offsprings inside [p1,p2]
    p1, p2 = parents
    cut_point = random.randint(key, (), 0, p1.shape[0])
    offspring1 = jnp.concatenate([p1[:cut_point], p2[cut_point:]])
    offspring2 = jnp.concatenate([p2[:cut_point], p1[cut_point:]])
    return offspring1, offspring2


def crossover(key, pref, parents):
    # crossover on a pair of parents -- (2, dim)
    return lax.cond(pref == 3,
                    extension_ray_crossover,
                    one_point_crossover,
                    key, parents
                    )


def batch_crossover(key, pref, parents):
    # crossover on multiple pairs of parents -- (n_pair, 2, dim)
    n_pair, _, _dim = parents.shape  # 1 n_pair is n_offspring / 2
    keys = random.split(key, n_pair)
    return vmap(crossover)(keys, pref, parents)


@jit_class
class GeneticAlgorithmPmin1(Algorithm):
    def __init__(self, pop_size, dim, mutation, p_c, p_m, tournament_size=3, tau_max=3):
        self.pop_size = pop_size
        self.dim = dim
        self.crossover = batch_crossover
        self.mutation = mutation
        self.n_offspring = self.pop_size  # we substitute all pop
        assert self.n_offspring % 2 == 0, "n_offspring must be even"
        self.tournament_size = tournament_size
        self.tau_max = tau_max
        self.selection = partial(full_tournament,
                                 tournament_size=self.tournament_size,
                                 tau_max=self.tau_max,
                                 minimization=True)  # Set to False if maximizing
        self.p_c = p_c
        self.p_m = p_m

    def setup(self, key):
        key, subkey = random.split(key)
        population = random.choice(subkey, 2, shape=(self.pop_size, self.dim)).astype(jnp.bool_)
        return State(
            contrib=jnp.ones((4,)) / 4,
            total_cross=jnp.zeros((4,), dtype=int),
            succ_cross=jnp.zeros((4,), dtype=int),
            population=population,
            parents=jnp.empty((self.n_offspring, self.dim), dtype=population.dtype),
            parents_index=jnp.empty((self.n_offspring,), dtype=int),  # what is this for in the general code?
            offspring=jnp.empty((self.n_offspring, self.dim), dtype=population.dtype),
            fitness=jnp.empty((self.pop_size,), dtype=int),
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

        is_eq = jnp.array([
            jnp.array_equal(offspring[0], parents[0]) | jnp.array_equal(offspring[0], parents[1]),
            jnp.array_equal(offspring[1], parents[0]) | jnp.array_equal(offspring[1], parents[1])
        ])
        # if any offspring satisfy: 1. not equal 2. better
        return jnp.any(~is_eq & is_better)

    def _update_variables(self, state, off_fit):
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
        contrib = state.contrib / state.total_cross
        total = jnp.sum(state.total_cross)
        contrib = lax.select(
            total == 0.0,
            jnp.ones((4,)) / 4,
            0.1 + (0.6 * (contrib / total))  # here contrib and total are arrays right? hence i am updating all 4 contribs not just 1
        )
        return state.replace(contrib=contrib)

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.replace(fitness=fitness)

    def ask(self, state):
        key, sel_key, cross_key, mut_key = random.split(state.key, 4)
        # choice a pref based on the current contribution
        # every 2 offsprings needs 1 pref
        pref = random.choice(state.key, 4, shape=(self.n_offspring // 2,), p=state.contrib)
        # select a batch of parents
        parents, parents_index = self.selection(sel_key, state.population, state.fitness, pref)
        # reshape into pairs of two, and do crossover on each pair
        offspring = self.crossover(cross_key, pref, parents)
        # mutation
        offspring = self.mutation(mut_key, offspring)
        new_state = state.replace(
            pref=pref,
            offspring=offspring,
            parents=parents,
            parents_index=parents_index,
            key=key
        )
        return offspring, new_state

    def tell(self, state, fitness):
        # contrib update
        state = self._update_variables(state, fitness)
        state = self._update_contrib(state)
        return state
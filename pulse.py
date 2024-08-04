from evox import Algorithm, Problem, State, jit_class, operators, workflows, monitors
import jax.numpy as jnp
from jax import vmap, lax, random


def extension_ray_crossover(key, parents):
    p1, p2 = parents
    # TODO
    return parents


def one_point_crossover(key, parents):
    p1, p2 = parents
    # TODO
    return parents


def crossover(key, perf, parents):
    # crossover on a pair of parents -- (2, dim)
    return lax.cond(perf == 3,
        extension_ray_crossover,
        one_point_crossover,
        key, parents
    )


def batch_crossover(key, perf, parents):
    # crossover on multiple pairs of parents -- (n_pair, 2, dim)
    n_pair, _, _dim = parents.shape
    keys = random.split(key, n_pair)
    return vmap(crossover)(keys, perf, parents)


@jit_class
class GeneticAlgorithmPmin1(Algorithm):
    def __init__(self, pop_size, dim, mutation, p_c, p_m):
        self.pop_size = pop_size
        self.dim = dim
        self.crossover = batch_crossover
        self.mutation = mutation
        self.n_offspring = self.pop_size // 2
        assert self.n_offspring % 2 == 0, "n_offspring must be even"
        self.selection = operators.selection.Tournament(self.n_offspring)
        self.p_c = p_c
        self.p_m = p_m

    def setup(self, key):
        key, subkey = random.split(key)
        population = random.choice(subkey, 2, shape=(self.pop_size, self.dim)).astype(jnp.bool_)
        return State(
            contrib=jnp.ones((4, )) / 4,
            total_cross=jnp.zeros((4, ), dtype=int),
            succ_cross=jnp.zeros((4, ), dtype=int),
            population=population,
            parents=jnp.empty((self.n_offspring, self.dim), dtype=population.dtype),
            parents_index=jnp.empty((self.n_offspring, ), dtype=int),
            offspring=jnp.empty((self.n_offspring, self.dim), dtype=population.dtype),
            fitness=jnp.empty((self.pop_size, ), dtype=int),
            perf=jnp.empty((self.n_offspring // 2, ), dtype=int),
            key=key
        )

    def _is_succ_cross(self, parents, offspring, par_fit, off_fit):
        # return True if it is a successful crossover
        # successful means one of the offspring satisfy the following:
        # 1. better than both parents
        # 2. different than both parents

        # parents, offspring --- (2, self.dim)
        # par_fit, off_fit ----- (2, )

        is_better = off_fit <= jnp.min(par_fit)
        is_eq = jnp.array([
            jnp.array_equal(off_fit[0], parents[0]) | jnp.array_equal(off_fit[0], parents[1]),
            jnp.array_equal(off_fit[1], parents[0]) | jnp.array_equal(off_fit[1], parents[1])
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
        succ_cross = state.succ_cross.at[state.perf].add(is_succ_cross)
        total_cross = state.total_cross.at[state.perf].add(1)
        return state.replace(succ_cross=succ_cross, total_cross=total_cross)


    def _update_contrib(self, state):
        contrib = state.contrib / state.total_cross
        total = jnp.sum(state.total_cross)
        contrib = lax.select(
            total == 0.0,
            jnp.ones((4, )) / 4,
            0.1 + (0.6 * (contrib / total))
        )
        return state.replace(contrib=contrib)

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.replace(fitness=fitness)

    def ask(self, state):
        key, sel_key, cross_key, mut_key = random.split(state.key, 4)
        # choice a perf based on the current contribution
        # every 2 offsprings needs 1 perf
        perf = random.choice(state.key, 4, shape=(self.n_offspring // 2,), p=state.contrib)
        # select a batch of parents
        parents, parents_index = self.selection(key, state.population, state.offspring)
        # reshape into pairs of two, and do crossover on each pair
        offspring = self.crossover(key, perf, parents.reshape(-1, 2, self.dim)).reshape(-1, self.dim)
        # mutation
        offspring = self.mutation(key, offspring)
        new_state = state.replace(
            perf=perf,
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
import numpy as np
from functools import reduce
import utils as uls
from random_search import RandomSearch
from solutions._point import _Point

''' 
SA selection with >= with 10% minimum pref3--> non-geometric ---> the logical gang
'''


class GeneticAlgorithmPmin1(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.minimization = problem_instance.minimization

        ''' Setting initial contributions for each preference. Paper had 4 preference levels.'''
        self.contrib_0 = 0.25
        self.contrib_1 = 0.25
        self.contrib_2 = 0.25
        self.contrib_3 = 0.25

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False):
        elite = self.best_solution
        elites = []
        contribs0 = [0.25]
        contribs1 = [0.25]
        contribs2 = [0.25]
        contribs3 = [0.25]

        for iteration in range(n_iterations):
            ''' Initialization of variables needed to compute Contribution each t generation'''
            self.succ_cross_0 = 0
            self.succ_cross_1 = 0
            self.succ_cross_2 = 0
            self.succ_cross_3 = 0

            self.total_cross_0 = 0
            self.total_cross_1 = 0
            self.total_cross_2 = 0
            self.total_cross_3 = 0

            offsprings = []

            while len(offsprings) < len(self.population):

                pref = self._pref_tournament()

                'Meta-heuristic crossover decider'
                if pref == 3:
                    self.crossover = uls.extension_ray_crossover
                else:
                    self.crossover = uls.one_point_crossover

                off1, off2 = p1, p2 = z1, z2 = self.selection(self.population, pref, self.problem_instance.minimization,
                                                              self._random_state)

                if self._random_state.uniform() < self.p_c:
                    if pref == 3:
                        off1, off2 = self._crossover_non_geometric(p1, p2)
                    else:
                        off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                self.problem_instance.evaluate(off1)
                self.problem_instance.evaluate(off2)

                '''
                Update cross success and total_cross
                z1,z2 are non updated parents aka OG b4 operations in order to compare offspring to parents
                '''
                self._update_variables(pref, off1, z1, off2, z2)

                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            self.population = offsprings
            elite = self._get_elite(self.population)
            elites.append(elite.fitness)

            '''Update Contributions'''
            self._contrib_update()
            contribs0.append(self.contrib_0)
            contribs1.append(self.contrib_1)
            contribs2.append(self.contrib_2)
            contribs3.append(self.contrib_3)

            if report:
                self._verbose_reporter_inner(elite, iteration)

        self.best_solution = elite
        uls.plot_fitness(elites, n_iterations,
                         'P_min_1_' + str(self.problem_instance.fitness_function).replace(" ", "_").replace('<',
                                                                                                            '').replace(
                             '>',
                             ''))
        uls.plot_contributions(contribs0, contribs1, contribs2, contribs3, n_iterations,
                               'P_min_1_' + str(self.problem_instance.fitness_function).replace(" ", "_").replace('<',
                                                                                                                  '').replace(
                                   '>', ''))
        self.best_elite_fitness = uls.get_elite(elites)

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = _Point(off1), _Point(off2)
        return off1, off2

    def _crossover_non_geometric(self, p1, p2):
        off1 = self.crossover(p1.representation, p2.representation, self._random_state)
        off2 = self.crossover(p2.representation, p1.representation, self._random_state)
        off1, off2 = _Point(off1), _Point(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = _Point(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def _pref_tournament(self):
        pref = self._random_state.choice([0, 1, 2, 3], 1,
                                         p=[self.contrib_0, self.contrib_1, self.contrib_2, self.contrib_3])
        return pref

    def _contrib_update(self):

        self.contrib_0 = self.succ_cross_0 / self.total_cross_0
        self.contrib_1 = self.succ_cross_1 / self.total_cross_1
        self.contrib_2 = self.succ_cross_2 / self.total_cross_2
        self.contrib_3 = self.succ_cross_3 / self.total_cross_3

        total = self.contrib_0 + self.contrib_1 + self.contrib_2 + self.contrib_3

        if total == 0:
            print('0 preferences had at least 1 success - Reset contributions')
            self.contrib_0 = 0.25
            self.contrib_1 = 0.25
            self.contrib_2 = 0.25
            self.contrib_3 = 0.25
        else:
            self.contrib_0 = 0.1 + (0.6 * (self.contrib_0 / total))
            self.contrib_1 = 0.1 + (0.6 * (self.contrib_1 / total))
            self.contrib_2 = 0.1 + (0.6 * (self.contrib_2 / total))
            self.contrib_3 = 0.1 + (0.6 * (self.contrib_3 / total))


    def _update_variables(self, pref, off1, z1, off2, z2):
        if self.minimization == 0:

            if pref == 0:
                if off1.fitness >= z1.fitness and off1.fitness >= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(off1.representation) != list(z2.representation):
                        self.succ_cross_0 += 1
                elif off2.fitness >= z1.fitness and off2.fitness >= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(off2.representation) != list(z2.representation):
                        self.succ_cross_0 += 1

                self.total_cross_0 += 1

            elif pref == 1:
                if off1.fitness >= z1.fitness and off1.fitness >= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(
                            off1.representation) != list(z2.representation):
                        self.succ_cross_1 += 1
                elif off2.fitness >= z1.fitness and off2.fitness >= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(
                            off2.representation) != list(z2.representation):
                        self.succ_cross_1 += 1

                self.total_cross_1 += 1

            elif pref == 2:
                if off1.fitness >= z1.fitness and off1.fitness >= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(
                            off1.representation) != list(z2.representation):
                        self.succ_cross_2 += 1
                elif off2.fitness >= z1.fitness and off2.fitness >= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(
                            off2.representation) != list(z2.representation):
                        self.succ_cross_2 += 1

                self.total_cross_2 += 1

            elif pref == 3:
                if off1.fitness >= z1.fitness and off1.fitness >= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(
                            off1.representation) != list(z2.representation):
                        self.succ_cross_3 += 1
                elif off2.fitness >= z1.fitness and off2.fitness >= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(
                            off2.representation) != list(z2.representation):
                        self.succ_cross_3 += 1

                self.total_cross_3 += 1

        else:

            if pref == 0:
                if off1.fitness <= z1.fitness and off1.fitness <= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(off1.representation) != list(z2.representation):
                        self.succ_cross_0 += 1
                elif off2.fitness <= z1.fitness and off2.fitness <= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(off2.representation) != list(z2.representation):
                        self.succ_cross_0 += 1

                self.total_cross_0 += 1

            elif pref == 1:
                if off1.fitness <= z1.fitness and off1.fitness <= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(
                            off1.representation) != list(z2.representation):
                        self.succ_cross_1 += 1
                elif off2.fitness <= z1.fitness and off2.fitness <= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(
                            off2.representation) != list(z2.representation):
                        self.succ_cross_1 += 1

                self.total_cross_1 += 1

            elif pref == 2:
                if off1.fitness <= z1.fitness and off1.fitness <= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(
                            off1.representation) != list(z2.representation):
                        self.succ_cross_2 += 1
                elif off2.fitness <= z1.fitness and off2.fitness <= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(
                            off2.representation) != list(z2.representation):
                        self.succ_cross_2 += 1

                self.total_cross_2 += 1

            elif pref == 3:
                if off1.fitness <= z1.fitness and off1.fitness <= z2.fitness:
                    if list(off1.representation) != list(z1.representation) and list(off1.representation) != list(z2.representation):
                        self.succ_cross_3 += 1
                elif off2.fitness <= z1.fitness and off2.fitness <= z2.fitness:
                    if list(off2.representation) != list(z1.representation) and list(off2.representation) != list(z2.representation):
                        self.succ_cross_3 += 1

                self.total_cross_3 += 1
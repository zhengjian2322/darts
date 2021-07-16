import random

from ConfigSpace.util import get_one_exchange_neighbourhood
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter


class AgingEvolution(object):
    def __init__(self, pop, cross_rate, mutation_rate, object_function, pop_size=100,
                 aging_sample_num=10):
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.object_function = object_function
        self.pop = np.array(pop.sample_configuration(self.pop_size))
        self.aging_sample_num = aging_sample_num
        self.history = np.zeros(self.pop_size)
        self.fitness(range(self.pop_size), 4)

    def mutate(self, child):
        neighbourhoods = get_one_exchange_neighbourhood(child, seed=0)
        return random.choice([n for n in neighbourhoods])

    def crossover(self, parent, pop):
        if random.random() > self.cross_rate:
            other_parent = random.choice(pop)
            for param_name in other_parent:
                if random.choice([True, False]):
                    parent[param_name] = other_parent[param_name]
        return parent

    def fitness(self, candidate_idx, epochs):
        for idx in candidate_idx:
            print('----idx', idx)
            self.history[idx] = self.object_function(self.pop[idx], epochs)

    def select(self):
        sample_population = np.random.choice(np.arange(self.pop_size), self.aging_sample_num)
        highest = np.max(self.history[sample_population])
        parent_dix = 0
        for individual in sample_population:
            if self.history[individual] == highest:
                parent_dix = individual
        return parent_dix

    def evolve(self):
        parent_idx = self.select()
        child = self.mutate(self.pop[parent_idx])
        self.pop = np.append(self.pop, [child], axis=0)
        with open('results.txt', 'a') as f:
            f.write(str(np.max(self.history)) + ' ' + ' '.join([str(h) for h in self.history]) + '\n')
        self.fitness([-1], 4)
        self.pop = self.pop[1:]

    def return_best_config(self):
        best_individual = np.where(self.history == np.max(self.history))
        return self.pop[best_individual][0]

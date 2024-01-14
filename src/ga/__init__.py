from abc import ABC

import numpy as np

from core.ca import CA


class GA(ABC):

    def __init__(self,
                 width,
                 height,
                 target,
                 population_size,
                 n_generations,
                 crossover_rate,
                 mutation_rate,
                 retention_rate,
                 steps):
        self.width = width
        self.height = height
        self.target = target
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.retention_size = int(self.population_size * retention_rate)
        self.steps = steps
        self.population = self.generate_population()
        self.population_fitness = self.get_population_fitness()

    def select(self):
        ind1 = np.random.randint(1, self.retention_size)
        ind2 = np.random.randint(1, self.retention_size)
        return ind1, ind2

    def get_population_fitness(self):
        return np.array([self.fitness_function(ind) for ind in self.population])

    def tournament(self, ind1, ind2):
        return (ind1, ind2) if self.population_fitness[ind1] > self.population_fitness[ind2] else (ind2, ind1)

    def evolution(self):
        for _ in range(self.n_generations):
            ranked_fitness_indices = np.argsort(self.population_fitness)
            new_population_indices = np.array([ranked_fitness_indices[0]])

            retained_samples = np.random.choice(ranked_fitness_indices[1:], size=self.retention_size - 1)
            new_population_indices = np.concatenate([new_population_indices, retained_samples])

            new_population_fitness = self.population_fitness[new_population_indices]
            new_population = [self.population[idx] for idx in new_population_indices]

            for _ in range(self.population_size - self.retention_size):
                ind_idx1, ind_idx2 = self.select()
                winner_idx, loser_idx = self.tournament(ind_idx1, ind_idx2)
                winner, loser = self.population[winner_idx], self.population[loser_idx]
                new_ind = self.crossover(winner, loser)
                new_ind = self.mutate(new_ind)
                new_population.append(new_ind)
                np.append(new_population_fitness, self.fitness_function(new_ind))

            self.population = new_population

    def get_top_n_fittest_individuals(self, n):
        ranked_fitness_indices = np.argsort(self.population_fitness)
        return [self.population[idx] for idx in ranked_fitness_indices[:n]]

    def generate_population(self) -> [CA]:
        pass

    def mutate(self, ind) -> CA:
        pass

    def crossover(self, winner, loser) -> CA:
        pass

    def fitness_function(self, individual) -> float:
        pass

from abc import ABC

import numpy as np

from core.ca import CA


class GA(ABC):

    def __init__(self,
                 target,
                 population_size,
                 n_generations,
                 crossover_rate,
                 mutation_rate,
                 retention_rate,
                 random_selection_rate,
                 steps,
                 fitness_function_name,
                 callback_function=None,
                 callback_interval=None):
        self.width = len(target)
        self.height = len(target[0])
        self.target = target
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.retention_size = int(self.population_size * retention_rate)
        self.random_selection_rate = random_selection_rate
        self.steps = steps
        self.fitness_function_name = fitness_function_name
        self.population = self.generate_population()
        self.population_fitness = self.get_population_fitness()
        self.best_fitness_over_generations = []
        self.callback_function = callback_function
        self.callback_interval = callback_interval

    def get_population_fitness(self):
        return np.array([self.fitness_function(ind) for ind in self.population])

    def tournament(self, ind1, ind2):
        return (ind1, ind2) if self.population_fitness[ind1] > self.population_fitness[ind2] else (ind2, ind1)

    def evolution(self):
        for generation in range(self.n_generations):
            self.evolution_step()
            print(f"Generation {generation}: Best fitness {self.get_best_fitness()}")
            if self.callback_function is not None and self.callback_interval is not None:
                if generation % self.callback_interval == 0:
                    self.callback_function(
                        generation,
                        self.get_fittest_individual(),
                        self.get_best_fitness(),
                        self.steps
                    )

    def evolution_step(self):
        ranked_fitness_indices = np.argsort(self.population_fitness)[::-1]

        # Take n best fit individuals
        retained_indices = ranked_fitness_indices[:self.retention_size].copy()
        leftover_indices = ranked_fitness_indices[self.retention_size:]

        # Give other individuals little chance to get selected
        for leftover_idx in leftover_indices:
            if np.random.rand() < self.random_selection_rate:
                retained_indices = np.append(retained_indices, leftover_idx)

        # Construct new population and fitness
        new_population_fitness = self.population_fitness[retained_indices].copy()
        new_population = [self.population[idx] for idx in retained_indices]

        # Mutate every individual except the best fit one
        for i in range(1, self.retention_size):
            new_population[i] = self.mutate(new_population[i])
            new_population_fitness[i] = self.fitness_function(new_population[i])

        # Store new population and new fitness scores
        self.population = new_population
        self.population_fitness = new_population_fitness

        retained_individuals_len = len(self.population)

        # For remaining positions
        for _ in range(self.population_size - retained_individuals_len):
            # Select two individuals
            ind_idx1, ind_idx2 = np.random.randint(0, retained_individuals_len - 1, 2)
            # Run a tournament
            winner_idx, loser_idx = self.tournament(ind_idx1, ind_idx2)
            # Select winner and loser
            winner, loser = self.population[winner_idx], self.population[loser_idx]
            # Perform crossover to create a new individual
            new_ind = self.crossover(winner, loser)

            # Append the new individual and it's fitness to the population
            self.population.append(new_ind)
            self.population_fitness = np.append(self.population_fitness, self.fitness_function(new_ind))

        best_fitness = max(self.population_fitness)
        self.best_fitness_over_generations.append(best_fitness)

    def get_top_n_fittest_individuals(self, n):
        ranked_fitness_indices = np.argsort(self.population_fitness)[::-1]
        return [
            {
                'individual': self.population[idx],
                'fitness': self.population_fitness[idx]
            } for idx in ranked_fitness_indices[:n]
        ]

    def get_best_fitness(self):
        return np.max(self.population_fitness)

    def get_best_fitness_over_generations(self):
        return self.best_fitness_over_generations

    def get_fittest_individual(self):
        max_fitness_idx = np.argmax(self.population_fitness)
        return self.population[max_fitness_idx]

    def generate_population(self) -> [CA]:
        pass

    def mutate(self, ind) -> CA:
        pass

    def crossover(self, winner, loser) -> CA:
        pass

    def fitness_function(self, individual: CA) -> float:
        pass

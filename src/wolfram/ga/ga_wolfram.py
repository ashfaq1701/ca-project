import numpy as np
from skimage.metrics import structural_similarity as ssim

from core.ca import CA
from ga import GA
from wolfram.ca import WolframCA, create_wolfram_ca_instance_from_pattern_and_rule_number
from wolfram.helpers import binary_string_to_binary_array, binary_array_to_number


class GAWolfram(GA):
    def __init__(self,
                 target,
                 population_size,
                 n_generations,
                 crossover_rate,
                 mutation_rate,
                 retention_rate,
                 random_selection_rate,
                 steps,
                 fitness_function_name="mae"):
        super().__init__(
            target,
            population_size,
            n_generations,
            crossover_rate,
            mutation_rate,
            retention_rate,
            random_selection_rate,
            steps,
            fitness_function_name
        )

    def generate_population(self):
        population = []

        for _ in range(self.population_size):
            rule_number = np.random.randint(0, 255)
            population.append(WolframCA(width=self.width, height=self.height, rule_number=rule_number))

        return population

    def mutate(self, ind):
        mutation_indices_states = np.array([np.random.rand() < self.mutation_rate for _ in range(self.width)])
        rule_indices_states = np.array([np.random.rand() < self.mutation_rate for _ in range(8)])

        states_copy = ind.board[0].copy()
        rule_array = binary_string_to_binary_array(ind.rule_binary)

        states_copy[mutation_indices_states] = (states_copy[mutation_indices_states] + 1) % 2
        rule_array[rule_indices_states] = (rule_array[rule_indices_states] + 1) % 2
        mutated_rule = binary_array_to_number(rule_array)

        return create_wolfram_ca_instance_from_pattern_and_rule_number(states_copy, mutated_rule)

    def crossover(self, winner, loser):
        crossover_indices_states = np.array([np.random.rand() < self.crossover_rate for _ in range(self.width)])
        crossover_indices_rules = np.array([np.random.rand() < self.crossover_rate for _ in range(8)])

        loser_states = loser.board[0]
        winner_states = winner.board[0]

        winner_rule_array = binary_string_to_binary_array(winner.rule_binary)
        loser_rule_array = binary_string_to_binary_array(loser.rule_binary)

        child_states = loser_states.copy()
        child_rule_array = loser_rule_array
        child_states[crossover_indices_states] = winner_states[crossover_indices_states]
        child_rule_array[crossover_indices_rules] = winner_rule_array[crossover_indices_rules]
        child_rule = binary_array_to_number(child_rule_array)

        return create_wolfram_ca_instance_from_pattern_and_rule_number(child_states, child_rule)

    def fitness_function(self, individual: CA):
        match self.fitness_function_name:
            case 'mae':
                return self.mae_fitness_function(individual)
            case 'structural_similarity':
                return self.structural_fitness_function(individual)
            case _:
                raise KeyError("Wrong fitness function name passed")

    def mae_fitness_function(self, individual: CA):
        n_genes = self.width * self.height
        resultant_board = individual.evolve(self.steps)
        return float(np.sum(resultant_board == self.target)) / float(n_genes)

    def structural_fitness_function(self, individual: CA):
        resultant_board = individual.evolve(self.steps)
        return ssim(resultant_board, self.target, data_range=1.0)

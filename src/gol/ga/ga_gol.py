import numpy as np
from skimage.metrics import structural_similarity as ssim
from core.ca import CA
from ga import GA
from gol.gol import GameOfLife, create_gol_instance_from_board


class GAGol(GA):
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
        return [GameOfLife(width=self.width, height=self.height) for _ in range(self.population_size)]

    def mutate(self, ind):
        mutation_indices = np.array([
            [np.random.rand() < self.mutation_rate for _ in range(self.width)]
            for _ in range(self.height)
        ])

        copied_board = ind.board.copy()
        copied_board[mutation_indices] = (copied_board[mutation_indices] + 1) % 2
        return create_gol_instance_from_board(copied_board)

    def crossover(self, winner, loser):
        cross_indices = np.array([
            [np.random.rand() < self.crossover_rate for _ in range(self.width)]
            for _ in range(self.height)
        ])

        loser_board = loser.board
        winner_board = winner.board

        child_board = loser_board.copy()
        child_board[cross_indices] = winner_board[cross_indices]
        return create_gol_instance_from_board(child_board)

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

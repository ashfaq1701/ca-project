import numpy as np
from ga import GA
from gol.gol import GameOfLife, create_gol_instance_from_board


class GAGol(GA):
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
        super().__init__(
            width,
            height,
            target,
            population_size,
            n_generations,
            crossover_rate,
            mutation_rate,
            retention_rate,
            steps
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

    def fitness_function(self, individual):
        n_genes = self.width * self.height
        return float(np.sum(individual.board == self.target)) / float(n_genes)

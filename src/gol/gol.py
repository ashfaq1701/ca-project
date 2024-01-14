import numpy as np

from core.ca import CA


def create_gol_instance_from_board(board):
    return GameOfLife(len(board), len(board[0]), init_method="specified", initial_states=board)


class GameOfLife(CA):
    def __init__(self, width, height, init_method="random", initial_states=None):
        super().__init__(width, height, init_method, initial_states)
        self.initialize(init_method, initial_states)

    def initialize(self, init_method, initial_states):
        match init_method:
            case "random":
                self.board = np.random.choice([0, 1], size=(self.height, self.width))
            case "specified":
                self.board = initial_states

    def evolve_step(self, step, current_state=None):
        if current_state is not None:
            board = current_state
        else:
            board = self.board.copy()

        for row in range(self.height):
            for col in range(self.width):
                neighbors = [
                    (row - 1, col),
                    (row - 1, col + 1),
                    (row, col + 1),
                    (row + 1, col + 1),
                    (row + 1, col),
                    (row + 1, col - 1),
                    (row, col - 1),
                    (row - 1, col - 1)
                ]

                alive_neighbor_count = 0

                for (nei_r, nei_c) in neighbors:
                    if 0 <= nei_r < self.height and 0 <= nei_c < self.width and abs(board[nei_r, nei_c]) == 1:
                        alive_neighbor_count += 1

                if board[row, col] == 1 and (alive_neighbor_count < 2 or alive_neighbor_count > 3):
                    board[row, col] = -1

                if board[row, col] == 0 and alive_neighbor_count == 3:
                    board[row, col] = 2

        for row in range(self.height):
            for col in range(self.width):
                if board[row, col] > 0:
                    board[row, col] = 1
                else:
                    board[row, col] = 0

        return board

    def evolve(self, steps):
        current_board = None

        for step in range(steps):
            current_board = self.evolve_step(step, current_board)

        return current_board

    def evolve_and_apply(self, steps):
        current_board = self.evolve(steps)
        self.apply(current_board)

    def apply(self, board):
        self.board = board

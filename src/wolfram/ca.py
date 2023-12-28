import numpy as np


class WolframCA:
    def __init__(
            self,
            width,
            height,
            rule_number: int,
            init_method="random",
            random_seed=42,
            initial_states=np.array([])
    ):
        self.width = width
        self.height = height
        self.rule_binary = format(rule_number, '08b')
        self.board = np.zeros(shape=(height, width), dtype=int)
        self.initialize(init_method, initial_states, random_seed)

    def initialize(self, init_method, initial_states, random_seed):
        match init_method:
            case "random":
                np.random.seed(random_seed)
                self.board[0] = np.random.choice([0, 1], size=self.width)
            case "center_one":
                row = np.zeros(self.width, dtype=int)
                row[int(self.width / 2)] = 1
                self.board[0] = row
            case "specified":
                self.board[0] = initial_states

    def evolve_step(self, generation):
        current_row_idx = generation if generation < self.height else self.height - 1
        current_states = self.board[current_row_idx]
        next_states = np.zeros(self.width, dtype=int)

        for col in range(self.width):
            left_neighbor = 0 if col == 0 else current_states[col - 1]
            right_neighbor = 0 if col == self.width - 1 else current_states[col + 1]
            neighbors = [left_neighbor, current_states[col], right_neighbor]
            next_state = self.get_next_state(neighbors)
            next_states[col] = next_state

        self.store_next_states(generation, next_states)

    def evolve(self, generations):
        for generation in range(generations):
            self.evolve_step(generation)

    def get_next_state(self, neighbors):
        rule_idx = abs(int(''.join(map(str, neighbors)), 2) - 7)
        return int(self.rule_binary[rule_idx])

    def store_next_states(self, generation, next_states):
        if generation < self.height - 1:
            self.board[generation + 1] = next_states
        else:
            self.shift_down(next_states)

    def shift_down(self, next_states):
        self.board = np.delete(self.board, (0,), axis=0)
        self.board = np.vstack([self.board, next_states])

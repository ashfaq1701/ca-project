import numpy as np

from core.ca import CA


class WolframCA(CA):
    def __init__(
            self,
            width,
            height,
            rule_number: int,
            init_method="random",
            initial_states=np.array([])
    ):
        super().__init__(width, height, init_method, initial_states)
        self.rule_binary = format(rule_number, '08b')
        self.board = np.zeros(shape=(height, width), dtype=int)
        self.initialize(init_method, initial_states)

    def initialize(self, init_method, initial_states):
        match init_method:
            case "random":
                self.board[0] = np.random.choice([0, 1], size=self.width)
            case "center_one":
                row = np.zeros(self.width, dtype=int)
                row[int(self.width / 2)] = 1
                self.board[0] = row
            case "specified":
                self.board[0] = initial_states

    def evolve_step(self, step):
        current_row_idx = step if step < self.height else self.height - 1
        current_states = self.board[current_row_idx]
        next_states = np.zeros(self.width, dtype=int)

        for col in range(self.width):
            left_neighbor = 0 if col == 0 else current_states[col - 1]
            right_neighbor = 0 if col == self.width - 1 else current_states[col + 1]
            neighbors = [left_neighbor, current_states[col], right_neighbor]
            next_state = self.get_next_state(neighbors)
            next_states[col] = next_state

        self.store_next_states(step, next_states)

    def evolve(self, steps):
        for step in range(steps):
            self.evolve_step(step)

    def get_next_state(self, neighbors):
        rule_idx = abs(int(''.join(map(str, neighbors)), 2) - 7)
        return int(self.rule_binary[rule_idx])

    def store_next_states(self, step, next_states):
        if step < self.height - 1:
            self.board[step + 1] = next_states
        else:
            self.shift_down(next_states)

    def shift_down(self, next_states):
        self.board = np.delete(self.board, (0,), axis=0)
        self.board = np.vstack([self.board, next_states])

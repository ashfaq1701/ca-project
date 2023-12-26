import numpy as np


class WolframCA:
    def __init__(
            self,
            width,
            steps,
            rule_number: int,
            init_method="random",
            random_seed=42,
            init_values=np.array([])
    ):
        self.width = width
        self.steps = steps
        self.rule_binary = format(rule_number, '08b')
        self.board = np.zeros(shape=(steps + 1, width), dtype=int)
        self.initialize(init_method, init_values, random_seed)

    def initialize(self, init_method, init_values, random_seed):
        match init_method:
            case "random":
                np.random.seed(random_seed)
                self.board[0] = np.random.choice([0, 1], size=self.width)
            case "center_one":
                row = np.zeros(self.width, dtype=int)
                row[int(self.width / 2)] = 1
                self.board[0] = row
            case "specified":
                self.board[0] = init_values

    def get_next_state(self, neighbors):
        rule_idx = abs(int(''.join(map(str, neighbors)), 2) - 7)
        return int(self.rule_binary[rule_idx])

    def evolve_step(self, step):
        current_states = self.board[step]
        next_states = self.board[step + 1]

        for col in range(self.width):
            left_neighbor = 0 if col == 0 else current_states[col - 1]
            right_neighbor = 0 if col == self.width - 1 else current_states[col + 1]
            neighbors = [left_neighbor, current_states[col], right_neighbor]
            next_state = self.get_next_state(neighbors)
            next_states[col] = next_state

    def evolve(self):
        for step in range(self.steps):
            self.evolve_step(step)

import numpy as np


class GameOfLife:
    def __init__(self, width, height, init_method="random", initial_state=None, random_seed=42):
        self.width = width
        self.height = height
        self.board = None
        self.initialize(init_method, initial_state, random_seed)

    def initialize(self, init_method, initial_state, random_seed):
        match init_method:
            case "random":
                np.random.seed(random_seed)
                self.board = np.random.choice([0, 1], size=(self.height, self.width))
            case "specified":
                self.board = initial_state


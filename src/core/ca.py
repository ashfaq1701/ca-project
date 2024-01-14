from abc import ABC


class CA(ABC):
    def __init__(self, width, height, init_method="random", initial_states=None):
        self.width = width
        self.height = height
        self.board = None

    def initialize(self, init_method, initial_states):
        pass

    def evolve_step(self, step):
        pass

    def evolve(self, steps):
        pass

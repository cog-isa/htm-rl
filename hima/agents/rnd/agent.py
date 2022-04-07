import numpy as np


class RndAgent:
    n_actions: int

    def __init__(self, n_actions, seed: int):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'rnd'

    def act(self):
        return self.rng.integers(self.n_actions)

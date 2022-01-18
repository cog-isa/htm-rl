import numpy as np

from htm_rl.agents.agent import Agent
from htm_rl.common.sdr import SparseSdr


class RndAgent(Agent):
    n_actions: int

    def __init__(
            self,
            n_actions: int,
            seed: int,
    ):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'rnd'

    def act(self, reward: float, state: SparseSdr, first: bool):
        return self.rng.integers(self.n_actions)

    def reset(self):
        pass

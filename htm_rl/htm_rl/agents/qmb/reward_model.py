import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay


class RewardModel:
    """
    Represents learned reward model on columns-level.
    """
    rewards: np.ndarray
    learning_rate: tuple[float, float]
    last_error: float

    def __init__(self, cells_sdr_size, learning_rate: tuple[float, float]):
        self.learning_rate = learning_rate
        self.rewards = np.zeros(cells_sdr_size, dtype=np.float)
        self.last_error = .0

    def update(self, s: SparseSdr, reward: float):
        lr = self.learning_rate[0]
        reward_estimate = self.state_reward(s)
        # reward_estimate = self.rewards[s]
        error = reward - reward_estimate

        self.rewards[s] += lr * error
        # self.last_error = error.mean()
        self.last_error = error

    def state_reward(self, s: SparseSdr):
        return np.median(self.rewards[s])

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)

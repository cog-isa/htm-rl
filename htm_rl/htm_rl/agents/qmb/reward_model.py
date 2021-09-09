import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_slice_lin_sum, exp_decay


class RewardModel:
    rewards: np.ndarray
    learning_rate: tuple[float, float]
    reward_anomaly: float

    def __init__(self, cells_sdr_size, learning_rate: tuple[float, float]):
        self.learning_rate = learning_rate
        self.rewards = np.zeros(cells_sdr_size, dtype=np.float)
        self.reward_anomaly = .0

    def update(self, s: SparseSdr, reward: float):
        learned_reward = self.rewards[s].mean()
        self.reward_anomaly = 2 * abs(learned_reward - reward) / (abs(learned_reward) + abs(reward) + 1e-5)
        update_slice_lin_sum(self.rewards, s, self.learning_rate[0], reward)

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)

import numpy as np

from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment


class RndAgent:
    n_actions: int

    def __init__(
            self,
            env: BioGwLabEnvironment,
            seed: int,
    ):
        self.n_actions = env.n_actions
        self.rng = np.random.default_rng(seed)

    @timed
    def run_episode(self, env):
        step = 0
        total_reward = 0.

        _, _, first = env.observe()
        while not (first and step > 0):
            action = self.rng.integers(self.n_actions)
            env.act(action)

            reward, _, first = env.observe()

            step += 1
            total_reward += reward

        return step, total_reward

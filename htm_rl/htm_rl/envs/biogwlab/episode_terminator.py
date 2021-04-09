from typing import Union

import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.entity import Entity
from htm_rl.envs.biogwlab.environment import Environment


class EpisodeTerminator:
    max_steps: int
    early_stop: bool

    terminated: bool

    _food: Entity

    def __init__(
            self, env: Environment,
            episode_max_steps: Union[int, str] = None,
            early_stop: bool = False,
    ):
        if isnone(episode_max_steps, 'auto') == 'auto':
            h, w = env.shape
            episode_max_steps = 2 * h * w

        self.max_steps = episode_max_steps
        self.early_stop = early_stop
        self._food = env.get_module('food')
        self.terminated = False

    def reset(self):
        self.terminated = False

    def collected(self):
        food = self._food
        if self.early_stop:
            # WARN: ugly workaround
            self.terminated = not np.any(food._rewards[food.map[food.mask]])

    def is_terminal(self, episode_step):
        if episode_step >= self.max_steps:
            self.terminated = True

        return self.terminated

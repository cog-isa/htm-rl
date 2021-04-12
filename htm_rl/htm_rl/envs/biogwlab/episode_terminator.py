from typing import Union

import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.biogwlab.module import Module, EntityType


class EpisodeTerminator(Module):
    max_steps: int
    early_stop: bool

    terminated: bool

    env: Environment

    def __init__(
            self, env: Environment, episode_max_steps: Union[int, str] = None,
            early_stop: bool = False,
            **module
    ):
        super().__init__(**module)

        if isnone(episode_max_steps, 'auto') == 'auto':
            h, w = env.shape
            episode_max_steps = 2 * h * w

        self.max_steps = episode_max_steps
        self.early_stop = early_stop
        self.terminated = False

    def reset(self):
        self.terminated = False

    def collected(self):
        if self.early_stop:
            # WARN: ugly workaround
            food = self.env.aggregated_map[EntityType.Consumable]
            self.terminated = not np.any(food)

    def is_terminal(self, episode_step):
        if episode_step >= self.max_steps:
            self.terminated = True

        return self.terminated

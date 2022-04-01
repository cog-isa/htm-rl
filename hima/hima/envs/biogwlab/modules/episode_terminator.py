from typing import Union

import numpy as np

from hima.common.utils import isnone
from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import Module, EntityType


class EpisodeTerminator(Module):
    max_steps: int
    early_stop: bool

    terminated: bool

    env: Environment

    def __init__(
            self, env: Environment, episode_max_steps: Union[int, str] = None,
            early_stop: bool = False, n_items_to_collect: int = None,
            **module
    ):
        super().__init__(**module)

        if isnone(episode_max_steps, 'auto') == 'auto':
            h, w = env.shape
            episode_max_steps = 2 * h * w

        self.env = env
        self.max_steps = episode_max_steps
        self.early_stop = early_stop
        self.terminated = False
        self.n_items_to_collect = n_items_to_collect

    def reset(self):
        self.terminated = False

    def collected(self):
        if self.early_stop:
            # WARN: ugly workaround
            food = self.env.aggregated_mask[EntityType.Consumable]
            self.terminated = not np.any(food)

            if self.n_items_to_collect is not None:
                self.terminated = self.env.items_collected == self.n_items_to_collect

    def is_terminal(self, episode_step):
        if episode_step >= self.max_steps:
            self.terminated = True

        return self.terminated

from typing import Tuple, Optional

import numpy as np

from htm_rl.envs.biogwlab.entity import Entity


class Agent(Entity):
    position: Tuple[int, int]
    view_direction: Optional[int]

    _obstacles: Entity
    _mask: np.ndarray

    def __init__(self, obstacles: Entity, **agent):
        super(Agent, self).__init__(**agent)
        self._obstacles = obstacles

    @property
    def mask(self):
        if self.mask is None:
            self._mask = np.zeros(self.shape, dtype=np.bool)
        self._mask[self.position] = True
        return self._mask

    @property
    def map(self):
        raise NotImplementedError

    def generate(self, seed):
        rnd = np.random.default_rng(seed)
        # HACK: to prevent spawning in food pos if there's just 1 food item of 1 type
        rnd.integers(1000, size=5)

        available_positions_mask = ~self._obstacles.mask
        available_positions_fl = np.flatnonzero(available_positions_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = self._unflatten_position(position_fl)

        self.position = position
        self.view_direction = rnd.choice(4)

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])
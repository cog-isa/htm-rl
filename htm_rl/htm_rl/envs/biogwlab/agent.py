from typing import Tuple, Optional

import numpy as np

from htm_rl.envs.biogwlab.entity import Entity
from htm_rl.envs.biogwlab.move_dynamics import MoveDynamics


class Agent(Entity):
    entity = 'agent'

    position: Tuple[int, int]
    view_direction: Optional[int]

    _obstacles: Entity
    _mask: np.ndarray

    def __init__(self, obstacles: Entity, **agent):
        super(Agent, self).__init__(**agent)
        self._obstacles = obstacles

    def generate(self, seed):
        rnd = np.random.default_rng(seed)
        # HACK: to prevent spawning in food pos if there's just 1 food item of 1 type
        rnd = np.random.default_rng(rnd.integers(100000))

        available_positions_mask = ~self._obstacles.mask
        available_positions_fl = np.flatnonzero(available_positions_mask)
        position_fl = rnd.choice(available_positions_fl)
        position = self._unflatten_position(position_fl)

        self.position = position
        self.view_direction = rnd.choice(4)

    def move(self, direction):
        self.position, success = MoveDynamics.try_move(
            self.position, direction, self.shape, self._obstacles.mask
        )

    def turn(self, turn_direction):
        self.view_direction = MoveDynamics.turn(self.view_direction, turn_direction)

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.shape[1])

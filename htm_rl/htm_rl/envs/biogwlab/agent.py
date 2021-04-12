from typing import Tuple, Optional

import numpy as np

from htm_rl.envs.biogwlab.module import Entity, EntityType
from htm_rl.envs.biogwlab.move_dynamics import MoveDynamics, DIRECTIONS_ORDER


class Agent(Entity):
    family = 'agent'
    type = EntityType.Agent

    position: Tuple[int, int]
    view_direction: Optional[int]

    env: 'Environment'

    def __init__(self, env: 'Environment', **entity):
        super(Agent, self).__init__(**entity)
        self.env = env

    def generate(self, seeds):
        rng = np.random.default_rng(seeds['agent'])
        empty_positions_mask = ~self.env.aggregated_map[EntityType.NonEmpty]

        empty_positions_fl = np.flatnonzero(empty_positions_mask)
        position_fl = rng.choice(empty_positions_fl)

        self.position = self._unflatten_position(position_fl)
        self.view_direction = rng.choice(len(DIRECTIONS_ORDER))

    def move(self, direction):
        obstacles = self.env.aggregated_map[EntityType.Obstacle]

        self.position, success = MoveDynamics.try_move(
            self.position, direction, self.env.shape, obstacles
        )

    def turn(self, turn_direction):
        self.view_direction = MoveDynamics.turn(self.view_direction, turn_direction)

    def append_mask(self, mask: np.ndarray):
        mask[self.position] = 1

    def append_position(self, exist: bool, position):
        return exist or self.position == position

    def _unflatten_position(self, flatten_position):
        return divmod(flatten_position, self.env.shape[1])
